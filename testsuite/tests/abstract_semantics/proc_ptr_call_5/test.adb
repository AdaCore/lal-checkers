procedure Test is

   X : Integer := 42;

   function F(E : Integer) return Integer is
   begin
      return E + 1;
   end F;

   function G(E : Integer) return Integer is
   begin
      return E + X;
   end G;

   P : access function (E : Integer) return Integer;
   B : Boolean;
   R : Integer;
begin
   if B then
      P := F'Access;
   else
      P := G'Access;
   end if;
   R := P(3);
end Test;
