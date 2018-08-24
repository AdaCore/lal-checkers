procedure Test is

   X : Integer := 42;

   function F(E : out Integer) return Boolean is
   begin
      E := 1;
      return False;
   end F;

   function G(E : out Integer) return Boolean is
   begin
      E := X;
      return True;
   end G;

   P : access function (E : out Integer) return Boolean;
   B : Boolean;
   R : Integer;
begin
   if B then
      P := F'Access;
   else
      P := G'Access;
   end if;
   B := P(R);
end Test;
