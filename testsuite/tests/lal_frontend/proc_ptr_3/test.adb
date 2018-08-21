procedure Main is
   X : Integer;
   P : access procedure;

   procedure G is
   begin
      X := 14;
   end G;

   procedure Test is
   begin
      P := G'Access;
   end Test;
begin
   Test;
end Main;
