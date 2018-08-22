procedure Test is
   X : Integer;

   type Proc_Access is access procedure;

   procedure F is
   begin
      X := 12;
   end F;

   function G return Proc_Access is
   begin
      return F'Access;
   end G;

   procedure Update (P : access procedure) is
   begin
      P.all;
   end Update;

begin
   Update(G);
end Test;
