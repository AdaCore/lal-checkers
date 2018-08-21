procedure Test is
   X : Integer;
   P : access procedure;

   procedure G is
   begin
      X := 14;
   end G;

   procedure F is
   begin
      P := G'Access;
   end F;
begin
   F;
end Test;
